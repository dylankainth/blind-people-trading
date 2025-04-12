import React from "react";
import {
  NavigationMenu,
  NavigationMenuList,
  NavigationMenuItem,
  NavigationMenuTrigger,
  NavigationMenuContent,
  NavigationMenuLink,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";
import Link from "next/link";

const Navbar: React.FC = () => {
  return (
    <nav className="flex items-center justify-between p-4 bg-white-800 text-black shadow-md">
      <img src="/trading-icon.svg" alt="Logo" className="h-10 w-10 mr-2" />
      <Link href="/" className="text-lg font-bold">
        MyApp
      </Link>
      <div className="flex-grow">
        <div className="flex items-center justify-end space-x-4">
          <NavigationMenu>
            <NavigationMenuList>
              <Link href="/docs" legacyBehavior passHref>
                <NavigationMenuLink className={navigationMenuTriggerStyle()}>
                  klnreslkrnesjirs
                </NavigationMenuLink>
              </Link>
            </NavigationMenuList>
          </NavigationMenu>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
